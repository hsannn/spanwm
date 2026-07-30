# nltk / wordnet live in the `spanwm` conda env -- call it by path
# PYTHON=/home/ssgyejin/miniconda3/envs/spanwm/bin/python
input_file="/home/ssgyejin/contents/spanwm/outputs/qwen_8b/sparkr_softfix_qwen3-8b_cnn_dailymail_n200.jsonl"
output_dir="outputs/qwen_8b/daily_mail/sparkr"
GPUS="0"


python paraphrasing_attack.py \
    --input "$input_file" \
    --output_dir "$output_dir" \
    --field watermarked_text \
    --model gpt-5-mini \
    --reasoning_effort low \
    --workers 8

python paraphrasing_attack.py \
    --input "$input_file" \
    --output_dir "$output_dir" \
    --field watermarked_text \
    --model openai/gpt-oss-20b \
    --gpus "$GPUS" \
    --reasoning_effort low \
    --batch_size 4

python paraphrasing_attack.py \
    --input "$input_file" \
    --output_dir "$output_dir" \
    --field watermarked_text \
    --model google/gemma-4-12B-it \
    --gpus "$GPUS" \
    --batch_size 4


# python paraphrasing_attack.py \
#     --input "/home/ssgyejin/contents/spanwm/outputs/qwen_4b/sweet_tau_qwen3-4b_c4_n200.jsonl" \
#     --output_dir "outputs/qwen_4b/c4/sweet" \
#     --field watermarked_text \
#     --model google/gemma-4-12B-it \
#     --gpus "$GPUS" \
#     --batch_size 4

# python paraphrasing_attack.py \
#     --input "/home/ssgyejin/contents/spanwm/outputs/qwen_4b/sweet_tau_qwen3-4b_wmt16_de_en_n200.jsonl" \
#     --output_dir "outputs/qwen_4b/wm16/sweet" \
#     --field watermarked_text \
#     --model google/gemma-4-12B-it \
#     --gpus "$GPUS" \
#     --batch_size 4
