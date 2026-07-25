"""Per-window-position green rate of a SpanWM v6/v7 output file, measured with
the detector-side scan. MUST run on the same device type as embedding (GPU):
torch.randperm streams differ between CPU and CUDA generators.
Also counts ' <punct>' splice artifacts."""
import argparse, json, os, re, sys
sys.path.insert(0, "/scratch2/sunny5574/spanwm")
os.chdir("/scratch2/sunny5574/spanwm")
import torch
from transformers import AutoTokenizer
from utils.transformers_config import TransformersConfig

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--version", choices=["v6", "v7"], required=True)
ap.add_argument("-n", type=int, default=40)
args = ap.parse_args()

if args.version == "v6":
    from watermark.spanwm_v6 import SpanWMV6 as WM
else:
    from watermark.spanwm_v7 import SpanWMV7 as WM

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {dev}  ({'OK: matches GPU embed' if dev == 'cuda' else 'WARNING: greenlists will NOT match a GPU embed'})")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
tc = TransformersConfig(model=None, tokenizer=tok, vocab_size=len(tok), device=dev)
wm = WM(f"config/SpanWM_{args.version}.json", tc)
K = wm.config.span_window_tokens

green = [0] * K; tot = [0] * K
n_done = n_art_pos = n_art_neg = n_dsp_pos = n_dsp_neg = n_all = 0
for line in open(args.input):
    r = json.loads(line); n_all += 1
    if re.search(r"\S+ [,.;:!?)]", r["watermarked_text"]): n_art_pos += 1
    if re.search(r"\S+ [,.;:!?)]", r.get("unwatermarked_text", "")): n_art_neg += 1
    if re.search(r"\S  +\S", r["watermarked_text"]): n_dsp_pos += 1
    if re.search(r"\S  +\S", r.get("unwatermarked_text", "")): n_dsp_neg += 1
    if n_done >= args.n:
        continue
    text = r["watermarked_text"]
    sites = wm.utils.scan_sites(text)
    if not sites:
        continue
    n_done += 1
    for sp in sites:
        pos, input_ids = wm.utils.mapper.window_positions(text, sp.start_char, K)
        ids = torch.as_tensor(input_ids)
        for j, idx in enumerate(pos):
            if idx < wm.config.prefix_length:
                continue
            gl = wm.utils.kgw.get_greenlist_ids(ids[:idx].to(dev))
            green[j] += int(ids[idx].to(dev) in gl); tot[j] += 1

print(f"samples scanned: {n_done}")
for j in range(K):
    rate = green[j] / tot[j] if tot[j] else float("nan")
    print(f"  pos {j}: green {rate:.3f}  (n={tot[j]})")
overall = sum(green) / sum(tot) if sum(tot) else float("nan")
print(f"  overall green rate: {overall:.3f}")
print(f"' <punct>' artifacts: watermarked {n_art_pos}/{n_all}   unwatermarked {n_art_neg}/{n_all}")
print(f"double-space artifacts: watermarked {n_dsp_pos}/{n_all}   unwatermarked {n_dsp_neg}/{n_all}")
