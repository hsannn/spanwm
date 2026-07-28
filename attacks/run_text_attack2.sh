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

input_file="/home/ssgyejin/contents/spanwm/outputs/ie_t2.2_c4_n200.jsonl"
output_dir="outputs/attacked/ie"


$PYTHON attacks/text_attack.py \
    --input "$input_file" \
    --output_dir "$output_dir" \
    --attacks WordDeletion SynonymSubstitution \
    --mode fixed \
    --ratios 0.1 0.2 0.3 0.4 0.5 \
    --field watermarked_text \
    --seed 42

# paraphrasing attack -- OpenAI API, needs OPENAI_API_KEY in ./.env (costs money)
$PYTHON attacks/paraphrasing_attack.py \
    --input "$input_file" \
    --output_dir "$output_dir" \
    --field watermarked_text \
    --model gpt-5-mini \
    --reasoning_effort low \
    --workers 8
