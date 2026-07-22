"""Diagnostic probe for SpanWM v6 (multi-span). Embed a few C4 prompts and
show per-site fills + pooled detection."""
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from watermark.spanwm_v6 import SpanWMV6
from utils.transformers_config import TransformersConfig

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
dev = "cuda" if torch.cuda.is_available() else "cpu"
M = "meta-llama/Llama-3.2-3B"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype="auto", device_map=dev).eval()
tc = TransformersConfig(model=model, tokenizer=tok, vocab_size=model.config.vocab_size,
                        device=dev, max_new_tokens=160, do_sample=True, top_p=0.9, temperature=0.8)
wm = SpanWMV6("config/SpanWM_v6.json", tc)

lines = open("dataset/c4/processed_c4.json").readlines()
for i in range(N):
    p = json.loads(lines[i])["prompt"]
    final = wm.generate_watermarked_text(p)
    info = wm.last_embedding_info
    print("=" * 92)
    print(f"[{i}] sites={info['num_sites']} aligned={info['num_aligned']} verified={info['verified']}")
    for s in info["sites"]:
        print(f"    {s['role'][0]}@{s['anchor']:<4} a{s['attempts']} d{s['anchor_dist']:<3} "
              f"{s['draft_span'][:28]!r} -> {s['fill'][:40]!r}")
    d = wm.detect_watermark(final)
    print(f"    DETECT sites={d['num_sites']} N={d['num_tested_tokens']} G={d['num_green_tokens']} "
          f"z={d['score']:.2f} p={d['p_value']:.2e} wm={d['is_watermarked']}")
