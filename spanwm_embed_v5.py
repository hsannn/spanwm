"""SpanWM v5 embedding: v3 recipe + per-span PRF role selection.

Same flow as spanwm_embed.py (v3) but the watermark site's role (S/O/P) is
derived per span from PRF(master_key, preceding n-gram), so it varies across
samples. v3/v4 files are untouched.

Run (GPU node, dlmwm env):
    python spanwm_embed_v5.py --dataset c4 --num_samples 200
"""

import argparse
import json
import os
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from watermark.spanwm_v5 import SpanWMV5
from utils.transformers_config import TransformersConfig
from evaluation.dataset import C4Dataset

MODEL_ID = "meta-llama/Llama-3.2-3B"
DATASETS = {"c4": (C4Dataset, "dataset/c4/processed_c4.json")}


def load_dataset(name, num_samples):
    if name not in DATASETS:
        raise ValueError(f"unknown dataset '{name}'. choices: {list(DATASETS)}")
    cls, path = DATASETS[name]
    return cls(path, max_samples=num_samples)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="c4", choices=list(DATASETS))
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--config", default="config/SpanWM_v5.json")
    ap.add_argument("--output", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    args = ap.parse_args()

    output = args.output or f"outputs/spanwm_v5_{args.dataset}_n{args.num_samples}.jsonl"
    os.makedirs(os.path.dirname(output), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  model={args.model}  dataset={args.dataset}  N={args.num_samples}", flush=True)

    ds = load_dataset(args.dataset, args.num_samples)
    n = min(args.num_samples, ds.prompt_nums)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto", device_map=device)
    model.eval()

    transformers_config = TransformersConfig(
        model=model, tokenizer=tokenizer, vocab_size=model.config.vocab_size,
        device=device, max_new_tokens=args.max_new_tokens,
        do_sample=True, top_p=0.9, temperature=0.8,
    )
    watermark = SpanWMV5(args.config, transformers_config)

    n_skipped = n_verified = 0
    role_counts = Counter()
    with open(output, "w") as fout:
        for i in range(n):
            prompt = ds.get_prompt(i)
            natural = ds.get_natural_text(i) if ds.natural_text_nums > i else ""

            wm_text = watermark.generate_watermarked_text(prompt)
            info = watermark.last_embedding_info
            unwm_text = watermark.generate_unwatermarked_text(prompt)

            skipped = info["skipped"]
            n_skipped += int(skipped)
            n_verified += int(info.get("verified", False))
            span = info["selected_span"]
            if span is not None:
                role_counts[span.role] += 1
            record = {
                "index": i,
                "prompt": prompt,
                "watermarked_text": wm_text,
                "unwatermarked_text": unwm_text,
                "natural_text": natural,
                "embed_skipped": skipped,
                "embed_verified": info.get("verified", False),
                "embed_attempts": info.get("attempts", 0),
                "embed_anchor_dist": info.get("anchor_dist"),
                "embed_role": None if span is None else span.role,
                "embed_span_text": None if span is None else span.text,
                "embed_reconstructed_span": info.get("reconstructed_span"),
                "embed_wm_char_range": info.get("wm_char_range"),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            if skipped:
                tag = "SKIP"
            else:
                v = "OK " if info.get("verified") else "UNV"
                tag = f"{v} a{info.get('attempts')} d{info.get('anchor_dist')} {span.role}:{span.text!r}"
            print(f"[{i + 1:>4}/{n}] {tag}", flush=True)

    print(f"\nwrote {n} records -> {output}   skipped: {n_skipped}   verified: {n_verified}/{n}", flush=True)
    print(f"role distribution: {dict(role_counts)}", flush=True)


if __name__ == "__main__":
    main()
