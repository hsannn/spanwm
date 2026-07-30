#!/bin/bash
# Rolling PPL evaluation on ONE A6000 while the grids run on the 3090s.
#
# Scans for completed cells not yet in the master summary, scores each batch
# with the established oracle protocol (gemma-4-12B BASE, vLLM prompt_logprobs,
# forced <bos>, char-level prompt strip; columns watermarked+unwatermarked),
# merges into outputs/ppl_scaleup.json, and exits once nothing is pending and
# no grid job remains.
#SBATCH -J ppl_watch
#SBATCH -p suma_a6000,gigabyte_a6000,tyan_a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 20:00:00
#SBATCH -o logs/%x-%j.out
set -u
export HF_HOME=/scratch/hyojun/hf_cache
export PATH="$HOME/anaconda3/envs/vllm/bin:$PATH"   # vLLM needs ninja on PATH
PY=$HOME/anaconda3/envs/vllm/bin/python
cd "$HOME/spanwm" || exit 1
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "PREFLIGHT FAIL: CUDA unavailable on $(hostname)"; exit 1; }
GB=$(ls -d /scratch/hyojun/hf_cache/hub/models--google--gemma-4-12B/snapshots/*/ | head -1)
mkdir -p outputs/ppl_batches
MASTER=outputs/ppl_scaleup.json

pending() {
  "$PY" - <<'PYEOF'
import glob, json, os
done = set()
if os.path.exists("outputs/ppl_scaleup.json"):
    done = set(json.load(open("outputs/ppl_scaleup.json")).get("files", {}))
out = []
# EXACT scope: the current 4-model x 3-scheme grid over c4 + continuation
# cnn/wmt16 (33 cells), llama3.2-3b/c4 excluded as already measured. Legacy
# summarization-cnn, German-prompt wmt16_de_en, gemma and SpanWM outputs are
# deliberately NOT here.
SCHEMES = ("sparkr_softfix", "sweet_tau", "ie_tau")
cands = []
for sc in SCHEMES:
    for mk in ("llama3.1-8b", "qwen3-4b", "qwen3-8b"):
        cands.append(f"outputs/scaleup/{sc}_{mk}_c4_n200.jsonl")
    for mk in ("llama3.2-3b", "llama3.1-8b", "qwen3-4b", "qwen3-8b"):
        cands.append(f"outputs/scaleup/{sc}_{mk}_cnn_n200.jsonl")
        cands.append(f"outputs/scaleup/{sc}_{mk}_wmt16_n200.jsonl")
cands = sorted(set(cands))
for p in cands:
    if p in done or not os.path.exists(p):
        continue
    if not os.path.exists(p.replace(".jsonl", ".unwatermarked.scores.json")):
        continue                       # cell not finished yet
    if sum(1 for _ in open(p)) < 200:
        continue
    out.append(p)
print("\n".join(out))
PYEOF
}

for round in $(seq 1 40); do
  LIST=$(pending)
  if [ -n "$LIST" ]; then
    N=$(echo "$LIST" | wc -l)
    echo "== round $round: $N개 파일 채점 =="
    B=outputs/ppl_batches/batch_${SLURM_JOB_ID}_${round}.json
    "$PY" baseline_ppl_vllm.py --input $LIST --oracle_model "$GB" \
          --which watermarked unwatermarked --truncation char --out "$B" || {
      echo "batch 실패 — 다음 라운드에 재시도"; sleep 300; continue; }
    "$PY" - "$B" <<'PYEOF'
import json, sys, os
b = json.load(open(sys.argv[1]))
m = {"files": {}}
if os.path.exists("outputs/ppl_scaleup.json"):
    m = json.load(open("outputs/ppl_scaleup.json"))
m.setdefault("oracle_model", b["oracle_model"])
m.setdefault("truncation", b["truncation"])
m["files"].update(b["files"])
json.dump(m, open("outputs/ppl_scaleup.json", "w"), indent=1)
print(f"master: {len(m['files'])} files")
PYEOF
  else
    Q=$(squeue -u "$USER" -h -o "%j" | grep -cE "_(cnn|wmt16)$")
    if [ "$Q" = "0" ]; then echo "== 대기 파일 없음 + 그리드 종료 -> 완료 =="; break; fi
    echo "== round $round: 신규 없음 (그리드 $Q job 진행 중), 10분 대기 =="
    sleep 600
  fi
done
echo "PPL WATCH DONE $(date)"
