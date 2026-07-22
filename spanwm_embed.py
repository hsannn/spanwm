"""SpanWM embedding: generate watermarked (span-regenerated) and
unwatermarked texts over a dataset, and dump them to jsonl.

Detection is a separate step (spanwm_detect.py) that re-parses the saved
text independently -- it never reads the embed-time span metadata below.

Run (GPU node, dlmwm env):
    python spanwm_embed.py --dataset c4 --num_samples 100
    python spanwm_embed.py --dataset c4 --num_samples 50 --output outputs/run1.jsonl
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from evaluation.dataset import C4Dataset

MODEL_ID = "meta-llama/Llama-3.2-3B"

# dataset name -> (loader class, path). Add new datasets here.
DATASETS = {
    "c4": (C4Dataset, "dataset/c4/processed_c4.json"),
}


def load_dataset(name: str, num_samples: int):
    if name not in DATASETS:
        raise ValueError(f"unknown dataset '{name}'. choices: {list(DATASETS)}")
    cls, path = DATASETS[name]
    return cls(path, max_samples=num_samples)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="c4", choices=list(DATASETS))
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--config", default="config/SpanWM.json")
    ap.add_argument("--output", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    args = ap.parse_args()

    output = args.output or f"outputs/spanwm_{args.dataset}_n{args.num_samples}.jsonl"
    os.makedirs(os.path.dirname(output), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  model={args.model}  dataset={args.dataset}  N={args.num_samples}")

    ds = load_dataset(args.dataset, args.num_samples)
    n = min(args.num_samples, ds.prompt_nums)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto", device_map=device)
    model.eval()

    transformers_config = TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        vocab_size=model.config.vocab_size,
        device=device,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
    )
    watermark = AutoWatermark.load(
        "SpanWM", algorithm_config=args.config, transformers_config=transformers_config)

    n_skipped = 0
    n_verified = 0
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
            record = {
                "index": i,
                "prompt": prompt,
                "watermarked_text": wm_text,
                "unwatermarked_text": unwm_text,
                "natural_text": natural,
                # embed-time metadata (for analysis only; detection ignores it)
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

    print(f"\nwrote {n} records -> {output}   skipped: {n_skipped}   verified: {n_verified}/{n}")


if __name__ == "__main__":
    main()
