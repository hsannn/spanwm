"""Diagnostic probe for SpanWM (base / left-AR + fixed-K). Runs the full embed
pipeline on a few C4 prompts and prints draft / anchor span / regenerated
window / verification / detection.

Usage (GPU node):
    srun --jobid=<ID> --overlap --chdir=/scratch2/sunny5574/spanwm \
        /home/sunny5574/miniconda3/envs/dlmwm/bin/python _probe_infill.py [N]
"""
import json, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig

MODEL = "meta-llama/Llama-3.2-3B"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype="auto", device_map=dev).eval()
tc = TransformersConfig(model=model, tokenizer=tok, vocab_size=model.config.vocab_size,
                        device=dev, max_new_tokens=160, do_sample=True, top_p=0.9, temperature=0.8)
wm = AutoWatermark.load("SpanWM", algorithm_config="config/SpanWM.json", transformers_config=tc)

lines = open("dataset/c4/processed_c4.json").readlines()
for i in range(N):
    prompt = json.loads(lines[i])["prompt"]
    final = wm.generate_watermarked_text(prompt)
    info = wm.last_embedding_info
    span = info["selected_span"]
    print("=" * 92)
    print(f"[{i}] PROMPT: {prompt[:70]!r}")
    if span is None:
        print("    SKIPPED (no eligible span in draft)"); continue
    ws, we = info["wm_char_range"]
    print(f"    anchor span (draft): role={span.role}  {span.text!r}")
    print(f"    VERIFY: verified={info['verified']} attempts={info['attempts']} anchor_dist={info['anchor_dist']}")
    print(f"    watermarked chars [{ws}:{we}] = {final[ws:we]!r}")
    print(f"    reconstructed span = {info['reconstructed_span']!r}")
    det = wm.detect_watermark(final)
    print(f"    DETECT: N={det['num_tested_tokens']} G={det['num_green_tokens']} "
          f"z={det['score']:.3f} p={det['p_value']:.2e} wm={det['is_watermarked']}")
