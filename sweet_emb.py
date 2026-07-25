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
    ap.add_argument("--config", default="config/SWEET.json")
    ap.add_argument("--output", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    args = ap.parse_args()

    output = args.output or f"outputs/sweet_{args.dataset}_n{args.num_samples}.jsonl"
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
        "SWEET", algorithm_config=args.config, transformers_config=transformers_config)

    with open(output, "w") as fout:
        for i in range(n):
            prompt = ds.get_prompt(i)
            natural = ds.get_natural_text(i) if ds.natural_text_nums > i else ""

            wm_text = watermark.generate_watermarked_text(prompt)
            unwm_text = watermark.generate_unwatermarked_text(prompt)

            record = {
                "index": i,
                "prompt": prompt,
                "watermarked_text": wm_text,
                "unwatermarked_text": unwm_text,
                "natural_text": natural,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{i + 1:>4}/{n}] done", flush=True)

    print(f"\nwrote {n} records -> {output}")


if __name__ == "__main__":
    main()
