import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from watermark.spanwm_v4 import SpanWMV4
from utils.transformers_config import TransformersConfig
dev="cuda" if torch.cuda.is_available() else "cpu"
M="meta-llama/Llama-3.2-3B"
tok=AutoTokenizer.from_pretrained(M)
model=AutoModelForCausalLM.from_pretrained(M,dtype="auto",device_map=dev).eval()
tc=TransformersConfig(model=model,tokenizer=tok,vocab_size=model.config.vocab_size,device=dev,
                      max_new_tokens=160,do_sample=True,top_p=0.9,temperature=0.8)
wm=SpanWMV4("config/SpanWM_v4.json",tc)
lines=open("dataset/c4/processed_c4.json").readlines()
for i in range(4):
    p=json.loads(lines[i])["prompt"]
    final=wm.generate_watermarked_text(p); info=wm.last_embedding_info; span=info["selected_span"]
    print("="*90); print(f"[{i}] anchor span={span.text!r}" if span else f"[{i}] SKIP")
    if span is None: continue
    ws,we=info["wm_char_range"]
    print(f"   fill (watermarked) = {final[ws:we]!r}")
    print(f"   verified={info['verified']} attempts={info['attempts']} recon={info['reconstructed_span']!r}")
    d=wm.detect_watermark(final)
    print(f"   DETECT N={d['num_tested_tokens']} G={d['num_green_tokens']} z={d['score']:.2f} p={d['p_value']:.2e} wm={d['is_watermarked']}")
