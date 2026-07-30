#!/bin/bash
# Measure the entropy distribution for one (model, dataset) and write the
# calibrated tau into outputs/entropy_stats.json.
#SBATCH -J caltau
#SBATCH -p base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 4:00:00
#SBATCH -o logs/%x-%j.out
set -u
MK=${1:?model key}; DS=${2:?dataset}
export HF_HOME=/scratch/hyojun/hf_cache
export TOKENIZERS_PARALLELISM=false
PY=$HOME/anaconda3/envs/wmattack/bin/python

# gemma-4 is a `gemma4_unified` checkpoint that transformers 4.55 refuses to
# load, so it runs in the `vllm` env (transformers 5.14) instead. Everything
# else stays on `wmattack`; the two are not interchangeable in either direction.
case "$MK" in
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

case "$MK" in
  llama3.2-3b) HFID="models--meta-llama--Llama-3.2-3B" ;;
  llama3.1-8b) HFID="models--meta-llama--Llama-3.1-8B" ;;
  qwen3-4b)    HFID="models--Qwen--Qwen3-4B-Base" ;;
  qwen3-8b)    HFID="models--Qwen--Qwen3-8B-Base" ;;
  gemma-4-12b)   HFID="models--google--gemma-4-12B" ;;
  *) echo "unknown model key: $MK"; exit 1 ;;
esac
MODEL=$(ls -d /scratch/hyojun/hf_cache/hub/$HFID/snapshots/*/ | head -1)
"$PY" measure_entropy.py --model "$MODEL" --model_key "$MK" --dataset "$DS" \
      --num_samples 200 --out outputs/entropy_stats.json
