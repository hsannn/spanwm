#!/bin/bash
# Scale-up cell: one (model x dataset x scheme) run -- generate 200 samples
# with per-sample timing + model metadata, then detect against both negatives.
#
#   sbatch scripts/run_scaleup.sbatch <MODEL_KEY> <DATASET> <SCHEME>
#     MODEL_KEY : llama3.2-3b | llama3.1-8b | qwen3-4b | qwen3-8b
#     DATASET   : c4 | cnn_dailymail | wmt16_de_en
#     SCHEME    : sparkr_softfix | sweet_tau | ie_tau
#
# tau note: SWEET and IE gate on the base model's next-token entropy, so
# the threshold is calibrated PER (model, dataset) as round(mean, 1) -- a
# fixed tau would gate a different fraction of tokens in every cell. Read
# from outputs/entropy_stats.json (scripts/calibrate_tau.sbatch).
#
# IE note: its entropy tagger is distilled from a SPECIFIC base model on a
# SPECIFIC domain (we measured a cross-provenance tagger at AUROC 0.45-0.50,
# i.e. chance), so this script points IE at
# watermark/ie/model/<MODEL_KEY>_<DATASET>/ and fails loudly if that tagger
# has not been distilled yet.
#SBATCH -J scaleup
#SBATCH -p base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
set -u
MODEL_KEY=${1:?model key}; DATASET=${2:?dataset}; SCHEME=${3:?scheme}
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

TAG="${SCHEME}_${MODEL_KEY}_${DATASET}"
OUT="outputs/scaleup/${TAG}_n200.jsonl"
mkdir -p outputs/scaleup

# ---- calibrated tau for this cell (SWEET / IE only) ----
TAU=""
if [ "$SCHEME" != "sparkr_softfix" ]; then
  TAU=$("$PY" tools/read_tau.py "$MODEL_KEY" "$DATASET")
  if [ -z "$TAU" ]; then
    echo "MISSING calibrated tau for ${MODEL_KEY}/${DATASET}"
    echo "  run first: sbatch scripts/calibrate_tau.sbatch $MODEL_KEY $DATASET"
    exit 1
  fi
  echo "calibrated tau(${MODEL_KEY}, ${DATASET}) = ${TAU}"
fi
mkdir -p config/generated

case "$SCHEME" in
  sparkr_softfix) ALG=SpARKR; CFG=config/SpARKR_softfix.json ;;
  sweet_tau)
      ALG=SWEET
      CFG="config/generated/SWEET_${MODEL_KEY}_${DATASET}.json"
      "$PY" tools/make_cfg.py sweet "$CFG" "$TAU"
      ;;
  ie_tau)
      ALG=IE
      TAGDIR="watermark/ie/model/${MODEL_KEY}_${DATASET}"
      TAGFILE="${TAGDIR}/entropy_tagger_$(echo "$TAU" | tr '.' '_').pt"
      if [ ! -f "$TAGFILE" ]; then
        echo "MISSING per-cell IE tagger: $TAGFILE"
        echo "  distill first: sbatch scripts/distill_ie.sbatch $MODEL_KEY $DATASET"
        exit 1
      fi
      CFG="config/generated/IE_${MODEL_KEY}_${DATASET}.json"
      "$PY" tools/make_cfg.py ie "$CFG" "$TAU" "$TAGDIR"
      ;;
  *) echo "unknown scheme: $SCHEME"; exit 1 ;;
esac

echo "== $TAG | node $(hostname) | model $MODEL | cfg $CFG | start $(date) =="
"$PY" baseline_embed.py --algorithm "$ALG" --config "$CFG" --model "$MODEL" \
      --dataset "$DATASET" --num_samples 200 --output "$OUT" || exit 1
for NEG in unwatermarked natural; do
  # collaborator wmt16 protocol carries no natural_text
  [ "$DATASET" = "wmt16" ] && [ "$NEG" = "natural" ] && continue
  "$PY" baseline_detect.py --input "$OUT" --algorithm "$ALG" --config "$CFG" \
        --model "$MODEL" --negative "$NEG"
done
echo "== $TAG done $(date) =="
