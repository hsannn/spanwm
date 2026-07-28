#!/bin/bash
# ================================================
# run_text_attack.sh
# Description: Run the word-level attacks (WordDeletion / SynonymSubstitution)
#              on a SpanWM output jsonl.
#
# Usage:  bash attacks/run_text_attack.sh
# ================================================

set -euo pipefail

# run from the project root so that the relative paths below work
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# nltk / wordnet live in the `spanwm` conda env -- call it by path
PYTHON=/home/ssgyejin/miniconda3/envs/spanwm/bin/python

# fixed ratios (10~50%) + random ratio per sample (10~50%)
$PYTHON attacks/text_attack.py \
    --input "outputs/spanwm_v7_c4_n200.jsonl" \
    --output_dir "outputs/attacked" \
    --attacks WordDeletion SynonymSubstitution \
    --mode random \
    --ratio_range 0.1 0.5 \
    --field watermarked_text \
    --seed 42

# --- variants -----------------------------------------------------------
# random ratio only
# $PYTHON attacks/text_attack.py \
#     --input "outputs/spanwm_v7_c4_n200.jsonl" \
#     --output_dir "outputs/attacked" \
#     --mode both \
#     --ratios 0.1 0.2 0.3 0.4 0.5 \
#     --ratio_range 0.1 0.5 \
#     --seed 42
#
