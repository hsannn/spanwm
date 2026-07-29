"""SpanWM v9 embedding: multi-span over benepar CONSTITUENTS (variable-length
sites; roles = NP/VP/PP labels).

Same flow as spanwm_embed_v7.py but uses SpanWMV9. v3-v8 files untouched.

Run (GPU node, spanwm env):
    python spanwm_embed_v9.py --dataset c4 --num_samples 200
"""

import argparse
import json
import os
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from watermark.spanwm_v9 import SpanWMV9
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
    ap.add_argument("--config", default="config/SpanWM_v9.json")
    ap.add_argument("--output", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    args = ap.parse_args()

    output = args.output or f"outputs/spanwm_v9_{args.dataset}_n{args.num_samples}.jsonl"
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
    watermark = SpanWMV9(args.config, transformers_config)

    n_skipped = n_verified = 0
    site_counts = Counter()
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
            site_counts[info["num_sites"]] += 1
            for s in info["sites"]:
                role_counts[s["role"]] += 1
            record = {
                "index": i,
                "prompt": prompt,
                "watermarked_text": wm_text,
                "unwatermarked_text": unwm_text,
                "natural_text": natural,
                "embed_skipped": skipped,
                "embed_verified": info.get("verified", False),
                "embed_num_sites": info["num_sites"],
                "embed_num_aligned": info["num_aligned"],
                "embed_sites": info["sites"],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            v = "SKIP" if skipped else ("OK " if info.get("verified") else "UNV")
            roles = "/".join(s["role"][0] for s in info["sites"])
            print(f"[{i + 1:>4}/{n}] {v} sites={info['num_sites']}({roles}) aligned={info['num_aligned']}", flush=True)

    print(f"\nwrote {n} records -> {output}   skipped: {n_skipped}   verified: {n_verified}/{n}", flush=True)
    print(f"sites per sample: {dict(sorted(site_counts.items()))}", flush=True)
    print(f"role distribution: {dict(role_counts)}", flush=True)
    print(f"parse failures (too-long sentences etc.): {watermark.utils.extractor.n_parse_failures}", flush=True)


if __name__ == "__main__":
    main()
