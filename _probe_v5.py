"""Diagnostic probe for SpanWM v5 (per-span PRF role). Full embed pipeline on a
few C4 prompts: chosen role varies per sample; verify + detect must line up."""
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from watermark.spanwm_v5 import SpanWMV5
from utils.transformers_config import TransformersConfig

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
dev = "cuda" if torch.cuda.is_available() else "cpu"
M = "meta-llama/Llama-3.2-3B"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype="auto", device_map=dev).eval()
tc = TransformersConfig(model=model, tokenizer=tok, vocab_size=model.config.vocab_size,
                        device=dev, max_new_tokens=160, do_sample=True, top_p=0.9, temperature=0.8)
wm = SpanWMV5("config/SpanWM_v5.json", tc)

lines = open("dataset/c4/processed_c4.json").readlines()
for i in range(N):
    p = json.loads(lines[i])["prompt"]
    final = wm.generate_watermarked_text(p)
    info = wm.last_embedding_info; span = info["selected_span"]
    print("=" * 92)
    if span is None:
        print(f"[{i}] SKIP"); continue
    print(f"[{i}] ROLE={span.role}  anchor span={span.text!r}")
    print(f"    verified={info['verified']} attempts={info['attempts']} anchor_dist={info['anchor_dist']}")
    print(f"    recon span = {info['reconstructed_span']!r}")
    d = wm.detect_watermark(final)
    print(f"    DETECT role={d['selected_role']} span={d['selected_span']!r}")
    print(f"    N={d['num_tested_tokens']} G={d['num_green_tokens']} z={d['score']:.2f} p={d['p_value']:.2e} wm={d['is_watermarked']}")
