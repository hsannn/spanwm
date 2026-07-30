#!/bin/bash
# Re-distill IE's entropy tagger for ONE (base model, dataset) pair.
#
#   sbatch scripts/distill_ie.sbatch <MODEL_KEY> <DATASET>
#
# Why per pair: the tagger's labels ARE the base model's next-token entropy on
# that domain. We measured the shipped taggers (StarCoder / MBPP code) at
# AUROC 0.45-0.50 -- chance -- on Llama-3.2-3B / C4, so a tagger may never be
# reused across a different base model or a different text domain. Training
# rows come from held-out documents (C4 rows 1000+; for CNN-DM / WMT16 the
# rows after the first 200 that the evaluation uses), so there is no overlap
# with anything reported.
#SBATCH -J distill
#SBATCH -p base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
set -u
MODEL_KEY=${1:?model key}; DATASET=${2:?dataset}
export HF_HOME=/scratch/hyojun/hf_cache
export TOKENIZERS_PARALLELISM=false
PY=$HOME/anaconda3/envs/wmattack/bin/python

# gemma-4 is a `gemma4_unified` checkpoint that transformers 4.55 refuses to
# load, so it runs in the `vllm` env (transformers 5.14) instead. Everything
# else stays on `wmattack`; the two are not interchangeable in either direction.
case "$MODEL_KEY" in
  gemma-4-12b) PY=$HOME/anaconda3/envs/vllm/bin/python ;;
esac

cd "$HOME/spanwm" || exit 1

# ---- preflight: refuse to run without a usable GPU ----
# torch silently falls back to CPU when CUDA init fails (we hit "CUDA unknown
# error" on a node and an 8B run went ~60x slower on CPU). That is not just
# slow: the KGW-family greenlist is torch.randperm(vocab), and CPU and CUDA
# give DIFFERENT permutations from the same seed, so a CPU-generated cell
# detected on a GPU reads as null. Fail loudly here instead.
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "PREFLIGHT FAIL: CUDA unavailable on $(hostname) -- refusing to run on CPU"
  echo "  (resubmit with --exclude=$(hostname) if the node stays broken)"
  exit 1
}


case "$MODEL_KEY" in
  llama3.2-3b) HFID="models--meta-llama--Llama-3.2-3B" ;;
  llama3.1-8b) HFID="models--meta-llama--Llama-3.1-8B" ;;
  qwen3-4b)    HFID="models--Qwen--Qwen3-4B-Base" ;;
  qwen3-8b)    HFID="models--Qwen--Qwen3-8B-Base" ;;
  gemma-4-12b)   HFID="models--google--gemma-4-12B" ;;
  *) echo "unknown model key: $MODEL_KEY"; exit 1 ;;
esac
MODEL=$(ls -d /scratch/hyojun/hf_cache/hub/$HFID/snapshots/*/ 2>/dev/null | head -1)
[ -z "$MODEL" ] && { echo "model not cached: $HFID"; exit 1; }

case "$DATASET" in
  c4)            DATA=dataset/c4/processed_c4.json; START=1000 ;;
  # the CNN/DM evaluation file IS the 200 reported rows -- nothing follows
  # row 200 to train on, so distillation reads a disjoint train-split file
  # built by tools/build_cnndm_heldout.py
  cnn_dailymail) DATA=dataset/cnn_dailymail/heldout_cnn_dailymail.json; START=0 ;;
  # continuation protocol: same held-out articles, article text only
  cnn)           DATA=dataset/cnn_dailymail/heldout_cnn_dailymail.json; START=0; EXTRA=--no_highlights ;;
  wmt16_de_en)   DATA=dataset/wmt16_de_en/processed_wmt16_de_en.json; START=200 ;;
  # collaborator protocol: the filtered eval set reaches raw row ~223, so
  # held-out training starts at 250 to stay disjoint with margin
  wmt16)         DATA=dataset/wmt16_de_en/processed_wmt16_de_en.json; START=250 ;;
  *) echo "unknown dataset: $DATASET"; exit 1 ;;
esac

EXTRA=${EXTRA:-}
TAU=$("$PY" tools/read_tau.py "$MODEL_KEY" "$DATASET")
if [ -z "$TAU" ]; then
  echo "MISSING calibrated tau for ${MODEL_KEY}/${DATASET}"
  echo "  run first: sbatch scripts/calibrate_tau.sbatch $MODEL_KEY $DATASET"
  exit 1
fi
echo "distilling at calibrated tau=${TAU}"
OUTDIR="watermark/ie/model/${MODEL_KEY}_${DATASET}"
mkdir -p "$OUTDIR"
echo "== distill $MODEL_KEY / $DATASET | rows ${START}+ | start $(date) =="
"$PY" train_ie_tagger.py --model "$MODEL" --data "$DATA" --start_row "$START" \
      --n_docs 5000 --max_pos 100 --target_positions 30000 \
      --taus "$TAU" --outdir "$OUTDIR" $EXTRA
echo "== distill $MODEL_KEY / $DATASET done $(date) =="
