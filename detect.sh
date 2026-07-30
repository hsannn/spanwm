python baseline_detect.py \
    --input "/home/ssgyejin/contents/spanwm/attacks/outputs/llama_8b/wm16/sparkr/spanwm_v8_Llama-3.1-8B_wmt16_n200_GPTParaphrase.jsonl" \
    --algorithm SpARKR \
    --config config/SpARKR.json \
    --column "attacked_text" \
    --truncation none \
    --model "meta-llama/Llama-3.1-8B" \
    --gpu 0

python baseline_detect.py \
    --input "/home/ssgyejin/contents/spanwm/attacks/outputs/llama_8b/wm16/sparkr/spanwm_v8_Llama-3.1-8B_wmt16_n200_Paraphrase_gemma-4-12B-it.jsonl" \
    --algorithm SpARKR \
    --config config/SpARKR.json \
    --column "attacked_text" \
    --truncation none \
    --model "meta-llama/Llama-3.1-8B" \
    --gpu 0

python spanwm_detect_v8.py \
    --input "//home/ssgyejin/contents/spanwm/attacks/outputs/c4/spanwm/spanwm_v8_c4_n200_Paraphrase_gemma-4-12B-it.jsonl" \
    --column "attacked_text" \
    --gpu 3
    
python spanwm_detect_v8.py \
    --input "/home/ssgyejin/contents/spanwm/attacks/outputs/c4/spanwm/spanwm_v8_c4_n200_Paraphrase_gpt-oss-20b.jsonl" \
    --column "attacked_text" \
    --gpu 3

