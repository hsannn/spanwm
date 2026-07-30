#!/bin/bash
# Mirror every finished scale-up cell into the local Downloads folder and print
# the results table. Safe to run repeatedly.
#
# The local copy is a MIRROR of the cluster, not an append-only pile: a cell
# that was superseded there (e.g. quarantined for running at the wrong tau)
# disappears here too, so a stale file can never back a reported row.
#
#   spanwm_scaleup/<model>/<dataset>/<scheme>_<model>_<dataset>_n200.{jsonl,
#                                     meta.json,*.scores.json}
DEST=~/Downloads/spanwm_scaleup
STAGE="$DEST/.staging"
mkdir -p "$STAGE"

# exact mirror of the cluster's canonical cells (_-prefixed dirs are archives)
rsync -az --delete --exclude '_*/' \
  hyojun@165.132.142.207:spanwm/outputs/scaleup/ "$STAGE/" 2>/dev/null || {
    echo "cluster unreachable -- reporting from the existing local mirror"; }
rsync -az hyojun@165.132.142.207:spanwm/outputs/entropy_stats.json "$DEST/" 2>/dev/null

# rebuild the <model>/<dataset>/ tree from the mirror.
# ONLY complete cells are laid out: a cell whose detection has not run yet has
# a .jsonl but no .scores.json, produces no table row, and would otherwise sit
# in the folder looking like a finished result. Those stay in .staging until
# they finish, so what is in the folder and what is in the table always agree.
rm -rf "$DEST"/llama3.* "$DEST"/qwen3-* "$DEST/_legacy" "$DEST/outputs" "$DEST/_incomplete"
inprog=""
for f in "$STAGE"/*.jsonl; do
  [ -f "$f" ] || continue
  b=$(basename "$f" .jsonl)
  if [[ ! $b =~ ^(sparkr_softfix|sweet_tau|ie_tau)_(llama3\.2-3b|llama3\.1-8b|qwen3-4b|qwen3-8b|gemma-4-12b)_(c4|cnn_dailymail|cnn|wmt16_de_en|wmt16)_n[0-9]+$ ]]; then
    mkdir -p "$DEST/_legacy" && cp -a "$STAGE/$b".* "$DEST/_legacy/" 2>/dev/null
    continue
  fi
  if [ ! -f "$STAGE/$b.unwatermarked.scores.json" ] || [ ! -f "$STAGE/$b.natural.scores.json" ]; then
    # baseline_embed.py opens its output up front and flushes every sample, so
    # the file exists from sample 1 -- its presence says nothing about being
    # finished. Report the actual line count and park it visibly: never hidden,
    # never in a folder whose name implies more progress than there is.
    mkdir -p "$DEST/_incomplete" && cp -a "$STAGE/$b".* "$DEST/_incomplete/"
    inprog="$inprog $b:$(wc -l < "$STAGE/$b.jsonl" | tr -d ' ')"
    continue
  fi
  d="$DEST/${BASH_REMATCH[2]}/${BASH_REMATCH[3]}"
  mkdir -p "$d" && cp -a "$STAGE/$b".* "$d/"
done
if [ -n "$inprog" ]; then
  echo "미완료 셀 (_incomplete/) -- 생성 진행도:"
  for c in $inprog; do
    n=${c##*:}; b=${c%:*}
    if [ "$n" -ge 200 ]; then echo "  $b  200/200 (생성 완료, 검출 대기)"
    else echo "  $b  $n/200 (생성 중)"; fi
  done
  echo
fi

python3 "$(dirname "$0")/harvest_scaleup.py" "$DEST" | tee "$DEST/RESULTS_TABLE.md"
python3 "$(dirname "$0")/timing_report.py"   "$DEST" > "$DEST/TIMING_TABLE.md"
